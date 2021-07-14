<template>
<div>
  <el-table
    v-if="son_props.flag === '1'"
    :data="son_props.data"
    style="width: 100%"
  >
    <el-table-column prop="notice_date" label="日期" sortable width="180"></el-table-column>
    <el-table-column prop="dis_time" label="公告" sortable width="200"></el-table-column>
    <el-table-column prop="col_type" label="类型" width="180"></el-table-column>
    <el-table-column prop="title" label="标题"></el-table-column>  
  </el-table>
  <el-table
    v-else-if="son_props.flag === '2'"
    :data="son_props.data"
    style="width: 100%"
  >
    <el-table-column
        width="180" 
        v-for=" i in son_props.th"
        :key="i.id"
        :property="i.prop"
        :label="i.propName">
    </el-table-column> 
  </el-table>
</div>
</template>
 
<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "ElTableSon",
  props: {
    son_props: {},
  },
  
});
</script>