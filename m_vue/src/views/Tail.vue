<template>
  <el-container>
    <el-aside width="200px">
      <el-row
        @click="stock_detail(ite)"
        type="flex"
        class="row-bg"
        justify="start"
        v-for="ite in lis"
        :key="ite.id"
      >{{ ite }}</el-row>
    </el-aside>
    <el-main>
      <el-row><ElTableSon :son_props="d.sina"/></el-row>
    </el-main>
  </el-container>
</template>

<script lang="ts">
import { defineComponent, ref, reactive } from "vue";
import ElTableSon from "@/components/ElTableSon.vue";
import axios from "axios";

export default defineComponent({
  name: "Tail",
  components: {
    ElTableSon,
  },
  setup() {
    const d = reactive({
      sina: {}
    });
    const lis = ref();
    axios
      .get("http://127.0.0.1:8000/datatables/stock_details/")
      .then(response => {
        lis.value = response.data;
      })
      .catch(error => {
        lis.value = [error];
      });
    function stock_detail(st: string) {
      console.log(st);
      axios
        .get("http://127.0.0.1:8000/datatables/stock_details/", {
          params: { st: st }
        })
        .then(response => {
          let r = response.data;
          //console.log(r.st_notice)
          /*                */
          d.sina = {
            flag: "2",
            data: r.sina,
            th: [
              {prop: "0", propName: "日期"},
              {prop: "1", propName: "时间"},
              {prop: "2", propName: "名称"},
              {prop: "3", propName: "现价"},
              {prop: "4", propName: "成交量"},
              {prop: "5", propName: "成交额"},
            ]
          };
        })
        .catch(error => {
          console.log(error);
        });
    }
    return { lis, stock_detail, d };
  }
});
</script>
