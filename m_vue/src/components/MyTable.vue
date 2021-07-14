<template>
<div v-if="parent_value.flag === 'flag'">
    <table border="0.5">
        <tr><th v-for="item in parent_value.tab_th" :key="item.id" >{{ item }}</th></tr>
        <tr v-for="ite in lis" :key="ite.id" @click="stock_detail(ite)">{{ ite }}</tr>
    </table>
</div>
<div v-else-if="parent_value.flag === 'only'">
    <table border="0.5" v-for="item in parent_value.data" :key="item.id">
        <caption><h5>{{ item.rq }} / {{ parent_value.cap }}</h5></caption>
        <div v-if="item.sdgd">
            <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
            <tr v-for="ite in item.sdgd" :key="ite.id">
                <td  >{{ ite.gdmc }}</td>
                <td  >{{ ite.gflx }}</td>
                <td  >{{ ite.zltgbcgbl }}</td>
                <td  >{{ ite.zj }}</td>
                <td  >{{ ite.bdbl }}</td>
            </tr>
        </div>
        <div v-if="item.sdltgd">
            <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
            <tr v-for="ite in item.sdltgd" :key="ite.id">
                <td  >{{ ite.gdmc }}</td>
                <td  >{{ ite.gdxz }}</td>
                <td  >{{ ite.gflx }}</td>
                <td  >{{ ite.zltgbcgbl }}</td>
                <td  >{{ ite.zj }}</td>
                <td  >{{ ite.bdbl }}</td>
            </tr>
        </div>
        <div v-if="parent_value.flag2 === 'institution_position'">
            <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
            <tr v-for="ite in item" :key="ite.id">
                <td  >{{ ite.rq }}</td>
                <td  >{{ ite.jglx }}</td>
                <td  >{{ ite.ccjs }}</td>
                <td  >{{ ite.zltgbl }}</td>
                <td  >{{ ite.zltgbbl }}</td>
            </tr>
        </div>
    </table>
</div>
<div v-else-if="parent_value.flag === 'lift_ban'">
    <table border="0.5">
        <caption><h5>{{ parent_value.cap }}</h5></caption>
        <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
        <tr v-for="ite in parent_value.data" :key="ite.id">
            <td  >{{ ite.jjsj }}</td>
            <td  >{{ ite.jjgzzgbbl }}</td>
            <td  >{{ ite.jjgzltgbbl }}</td>
            <td  >{{ ite.gplx }}</td>
        </tr>
    </table>
</div>
<div v-else-if="parent_value.flag === 'share_num'">
    <table border="0.5">
        <caption><h5>{{ parent_value.cap }}</h5></caption>
        <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
        <tr v-for="ite in parent_value.data" :key="ite.id">
            <td >{{ ite.notice_dat }}</td>
            <td >{{ ite.end_da }}</td>
            <td >{{ ite.add_rate }}</td>
            <td >{{ ite.total }}</td>
            <td >{{ ite.share_num }}</td>
            <td >{{ ite.change_share }}</td>
            <td >{{ ite.change_reason }}</td>
        </tr>
    </table>
</div>
<div v-else-if="parent_value.flag === 'bai'">
    <table border="0.5">
        <caption><h5>{{ parent_value.cap }}</h5></caption>
        <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
        <tr v-for="ite in parent_value.data" :key="ite.id">
            <td >{{ ite.time }}</td>
            <td >{{ ite.title }}</td>
            <td >{{ ite.des }}</td>
        </tr>
    </table>
</div>
<div v-else-if="parent_value.flag === 'circle_table'">
    <table border="0.5" v-for="item in parent_value.data" :key="item.id">
        <caption><h5>{{ parent_value.cap }}</h5></caption>
        <tr><th v-for="m in parent_value.tab_th" :key="m.id" >{{ m }}</th></tr>
        <tr v-for="ite in item" :key="ite.id"><td  v-for="t in ite" :key="t.id">{{ t }}</td></tr>
    </table>
</div>
<div v-else>
    <table border="0.5">
        <caption><h5>{{ parent_value.cap }}</h5></caption>
        <div v-if="parent_value.data">
            <tr><th v-for="item in parent_value.tab_th" :key="item.id" >{{ item }}</th></tr>
            <tr v-for="ite in parent_value.data" :key="ite.id"><td v-for="(value, name, index)  in ite" :key="index">{{ value  }}</td></tr>
        </div>
    </table>
</div>
</template>

<script lang="ts">
import { ref,toRefs, defineComponent  } from "vue";
import axios from 'axios';

export default defineComponent ({
    name: 'MyTable',
    props: {
        parent_value: Object,
    },
    setup(props:any, { emit }) {
        const { parent_value } = toRefs(props);
        const lis = ref();
        if (parent_value.value.url){
            axios.get(parent_value.value.url).then(response => {
                lis.value = response.data
                //console.log(lis.value);
            }).catch(error => {
                lis.value = [ error ]
            });
        }
        function stock_detail(st:string){
            emit('stock_code', st)
        }
        return { lis, stock_detail }
    },
})
</script>