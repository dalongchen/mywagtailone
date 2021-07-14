<template>
<table border="1">
    <tr>
        <td><MyTable :parent_value="d.table_th"  @stock_code="stock_code"/></td>
        <td>
            <MyTable :parent_value="d.sina"/>
            <MyTable :parent_value="d.finance"/>
            <MyTable :parent_value="d.st_ach"/>
            <MyTable :parent_value="d.per_for"/>
            <MyTable :parent_value="d.ten_share"/>
            <MyTable :parent_value="d.ten_current_share"/>
            <MyTable :parent_value="d.institution_position_son"/>
            <MyTable :parent_value="d.lift_ban"/>
            <MyTable :parent_value="d.lgt"/>
            <MyTable :parent_value="d.add_subtract"/>
            <MyTable :parent_value="d.manager_a"/>
            <MyTable :parent_value="d.rz"/>
            <MyTable :parent_value="d.capital_inflow"/>
            <MyTable :parent_value="d.institution_res"/>
            <MyTable :parent_value="d.share_num"/>
            <MyTable :parent_value="d.dragon_tiger"/>
            <MyTable :parent_value="d.ins_re_re"/>
            <MyTable :parent_value="d.xq_discuss"/>
            <MyTable :parent_value="d.xq_new"/>
            <MyTable :parent_value="d.bai"/>
            <MyTable :parent_value="d.st_notice"/>
        </td>
    </tr>
</table>
</template>

<script lang="ts">
import { defineComponent,reactive } from "vue";
import MyTable from '@/components/MyTable.vue'; // @ is an alias to /src
import axios from 'axios'

export default defineComponent ({
    name: 'About',
    components: {
        MyTable,
    },

    setup() {
        const d = reactive({
            table_th: {"flag": "flag", "tab_th":["股票"], "url": "http://127.0.0.1:8000/datatables/stock_details/"},
            stock_name: "",
            sina: {},
            finance: {},
            st_ach: {},
            per_for: {},
            ten_share: {},
            ten_current_share: {},
            institution_position_son: {},
            lift_ban: {},
            lgt: {},
            add_subtract: {},
            manager_a: {},
            rz: {},
            capital_inflow: {},
            institution_res: {},
            share_num: {},
            dragon_tiger: {},
            ins_re_re: {},
            xq_discuss: {},
            xq_new: {},
            bai: {},
            st_notice: {},
        })
        function stock_code (st:string) {
            console.log(st)
            d.stock_name = st
            axios.get("http://127.0.0.1:8000/datatables/stock_details/",{ params:{ st: st } }).then(response => {
                let r = response.data
                //console.log(r.st_notice)
                /*                */
                d.sina = {"cap": "新浪行情", "data": r.sina, "tab_th":["日期", "时间", "名称", "现价", "成交量", "成交额"]}
                d.finance = {"cap": "财务", "data": r.finance }
                d.st_ach = {"cap": "业绩", "data": r.st_ach, "tab_th":["日期", "截至", "收益", "扣非", "营收", "同比%","环比","利润","同比","环比","净资产","收益率","现金流","毛利","分配" ]}
                d.per_for = {"cap": "业绩预告(一年内)", "data": r.per_for, "tab_th":["报告期", "公告日", "指标","类型", "预测下", "预测上", "幅度下", "幅度上", "业绩", "原因"]}
                d.ten_share = {"flag": "only", "cap": "十大股东", "data": r.ins_pos.ten_share, "tab_th":["名称", "类型", "占总比","增减（股）", "变动比例"]}
                d.ten_current_share = {"flag": "only", "cap": "十大流通股东", "data": r.ins_pos.ten_current_share, "tab_th":["名称", "性质", "类型", "占总比","增减（股）", "变动比例"]}
                d.institution_position_son = {"flag": "only", "flag2": "institution_position", "cap": "机构持仓", "data": r.ins_pos.institution_position_son , "tab_th":["季度", "类型", "家数", "流通比","总比"]}
                d.lift_ban = {"flag": "lift_ban", "cap": "解禁", "data": r.ins_pos.lift_ban , "tab_th":["解禁时间", "总股本比", "流通比", "类型",]}
                d.lgt = {"cap": "陆股通", "data": r.lgt , "tab_th":["时间", "占总股比", "金额"]}
                d.add_subtract = {"cap": "股东减持（半年）", "data": r.add_subtract , "tab_th":["名称", "增减", "方式", "变动总比", "变动流通比", "剩股总股比","剩股流通比","开始日","截至日","公告日"]}
                d.manager_a = {"cap": "高管减持（1年）", "data": r.manager_a , "tab_th":["日期", "变动人", "变动数亿", "均价", "比例", "变动后亿","原因","类型","懂监高","关系"]}
                d.rz = {"cap": "融资融券(5天)", "data": r.rz , "tab_th":["日期", "融资余额", "占流通比", "净买入", "融券余额", "俩融差亿"]}
                d.capital_inflow = {"cap": "资金流入", "data": r.capital_inflow , "tab_th":["日期", "净流入亿"]}
                d.institution_res = {"cap": "机构调研(90天)", "data": r.institution_res , "tab_th":["公告日", "接待日", "数量", "方式"]}
                d.share_num = {"flag": "share_num", "cap": "股东人数", "data": r.share_num , "tab_th":["日期", "截至", "增减比", "总市值", "股东数", "变化股", "原因"]}
                d.dragon_tiger = {"flag": "circle_table", "cap": "龙虎榜", "data": r.dragon_tiger , "tab_th":["序号", "名称", "次数", "胜率", "买入额", "占比", "卖出额", "占比", "净额"]}
                d.ins_re_re = {"cap": "研究报告（半年内）", "data": r.ins_re_re , "tab_th":["日期", "标题"]}
                d.xq_discuss = {"cap": "雪球论坛", "data": r.xq_discuss , "tab_th":["日期", "标题", "描述"]}
                d.xq_new = {"cap": "雪球资信", "data": r.xq_new , "tab_th":["日期", "标题", "描述"]}
                d.bai = {"flag": "bai", "cap": "百度个股负面", "data": r.bai , "tab_th":["日期", "标题", "描述"]}
                d.st_notice = {"cap": "公告", "data": r.st_notice , "tab_th":["日期", "类型", "标题"]}
            }).catch(error => {
                console.log(error)
            });
        }
        return { d, stock_code }
    },
})

// npm install vue-class-component vue-property-decorator --save-dev
// npm uninstall vue-property-decorator --save-dev
</script>
